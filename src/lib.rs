/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! An implementation of thread-safe swap chains for the `surfman` surface manager.
//!
//! The role of a swap chain is to allow surfaces to be communicated between contexts,
//! often in different threads. Each swap chain has a *producer* context,
//! responsible for creating and destroying surfaces, and a number of *consumer* contexts,
//! (usually just one) which take surfaces from the swap chain, and return them for recycling.
//!
//! Each swap chain has a *back buffer*, that is the current surface that the producer context may draw to.
//! Each swap chain has a *front buffer*, that is the most recent surface the producer context finished drawing to.
//!
//! The producer may *swap* these buffers when it has finished drawing and has a surface ready to display.
//!
//! The consumer may *take* the front buffer, display it, then *recycle* it.
//!
//! Each producer context has one *attached* swap chain, whose back buffer is the current surface of the context.
//! The producer may change the attached swap chain, attaching a currently unattached swap chain,
//! and detatching the currently attached one.

use euclid::default::Size2D;

use fnv::FnvHashMap;
use fnv::FnvHashSet;

use log::debug;

use std::collections::hash_map::Entry;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

use sparkle::gl;
use sparkle::gl::GLuint;
use sparkle::gl::Gl;

use surfman::platform::generic::universal::context::Context;
use surfman::platform::generic::universal::device::Device;
use surfman::platform::generic::universal::surface::Surface;
use surfman::ContextID;
use surfman::Error;
use surfman::SurfaceAccess;
use surfman::SurfaceType;

// The data stored for each swap chain.
struct SwapChainData {
    // The size of the back buffer
    size: Size2D<i32>,
    // The id of the producer context
    context_id: ContextID,
    // The surface access mode for the context.
    surface_access: SurfaceAccess,
    // The back buffer of the swap chain.
    // Some if this swap chain is unattached,
    // None if it is attached (and so the context owns the surface).
    unattached_surface: Option<Surface>,
    // Some if the producing context has finished drawing a new front buffer, ready to be displayed.
    pending_surface: Option<Surface>,
    // All of the surfaces that have already been displayed, ready to be recycled.
    recycled_surfaces: Vec<Surface>,
}

impl SwapChainData {
    // Returns `Ok` if `context` is the producer context for this swap chain.
    fn validate_context(&self, device: &Device, context: &Context) -> Result<(), Error> {
        if self.context_id == device.context_id(context) {
            Ok(())
        } else {
            Err(Error::IncompatibleContext)
        }
    }

    // Swap the back and front buffers.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for this swap chain.
    fn swap_buffers(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        debug!("Swap buffers on context {:?}", self.context_id);
        self.validate_context(device, context)?;

        // Recycle the old front buffer
        if let Some(old_front_buffer) = self.pending_surface.take() {
            debug!(
                "Recycling surface {:?} ({:?}) for context {:?}",
                old_front_buffer.id(),
                old_front_buffer.size(),
                self.context_id
            );
            self.recycle_surface(old_front_buffer);
        }

        // Fetch a new back buffer, recycling presented buffers if possible.
        let new_back_buffer = self
            .recycled_surfaces
            .iter()
            .position(|surface| surface.size() == self.size)
            .map(|index| {
                debug!("Recyling surface for context {:?}", self.context_id);
                Ok(self.recycled_surfaces.swap_remove(index))
            })
            .unwrap_or_else(|| {
                debug!(
                    "Creating a new surface ({:?}) for context {:?}",
                    self.size, self.context_id
                );
                let surface_type = SurfaceType::Generic { size: self.size };
                device.create_surface(context, self.surface_access, &surface_type)
            })?;

        // Swap the buffers
        debug!(
            "Surface {:?} is the new back buffer for context {:?}",
            new_back_buffer.id(),
            self.context_id
        );
        let new_front_buffer = match self.unattached_surface.as_mut() {
            Some(surface) => {
                debug!("Replacing unattached surface");
                mem::replace(surface, new_back_buffer)
            }
            None => {
                debug!("Replacing attached surface");
                device.replace_context_surface(context, new_back_buffer)?
            }
        };

        // Update the state
        debug!(
            "Surface {:?} is the new front buffer for context {:?}",
            new_front_buffer.id(),
            self.context_id
        );
        self.pending_surface = Some(new_front_buffer);
        for surface in self.recycled_surfaces.drain(..) {
            debug!("Destroying a surface for context {:?}", self.context_id);
            device.destroy_surface(context, surface)?;
        }

        Ok(())
    }

    // Swap the attached swap chain.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for both swap chains.
    // Returns an error if this swap chain is attached, or the other swap chain is detached.
    fn take_attachment_from(
        &mut self,
        device: &mut Device,
        context: &mut Context,
        other: &mut SwapChainData,
    ) -> Result<(), Error> {
        self.validate_context(device, context)?;
        other.validate_context(device, context)?;
        if let (Some(surface), true) = (
            self.unattached_surface.take(),
            other.unattached_surface.is_none(),
        ) {
            debug!("Attaching surface {:?}", surface.id());
            let surface = device.replace_context_surface(context, surface)?;
            debug!("Detaching surface {:?}", surface.id());
            other.unattached_surface = Some(surface);
            Ok(())
        } else {
            Err(Error::Failed)
        }
    }

    // Resize the swap chain.
    // This creates a new back buffer of the appropriate size,
    // and destroys the old one.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for this swap chain.
    // Returns an error if `size` is smaller than (1, 1).
    fn resize(
        &mut self,
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        debug!("Resizing context {:?} to {:?}", device.context_id(context), size);
        self.validate_context(device, context)?;
        if (size.width < 1) || (size.height < 1) {
            return Err(Error::Failed);
        }
        let surface_type = SurfaceType::Generic { size };
        let new_back_buffer = device.create_surface(context, self.surface_access, &surface_type)?;
        debug!(
            "Surface {:?} is the new back buffer for context {:?}",
            new_back_buffer.id(),
            self.context_id
        );
        let old_back_buffer = match self.unattached_surface.as_mut() {
            Some(surface) => mem::replace(surface, new_back_buffer),
            None => device.replace_context_surface(context, new_back_buffer)?,
        };
        device.destroy_surface(context, old_back_buffer)?;
        self.size = size;
        Ok(())
    }

    // Take the current front buffer.
    // Called by a consumer.
    fn take_surface(&mut self) -> Option<Surface> {
        self.pending_surface
            .take()
            .or_else(|| self.recycled_surfaces.pop())
    }

    // Recycle the current front buffer.
    // Called by a consumer.
    fn recycle_surface(&mut self, surface: Surface) {
        self.recycled_surfaces.push(surface)
    }

    // Clear the current back buffer.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for this swap chain.
    fn clear_surface(
        &mut self,
        device: &mut Device,
        context: &mut Context,
        gl: &Gl,
    ) -> Result<(), Error> {
        self.validate_context(device, context)?;

        // Save the current GL state
        let mut bound_fbos = [0, 0];
        let mut clear_color = [0., 0., 0., 0.];
        let mut clear_depth = [0.];
        let mut clear_stencil = [0];
        let scissor_enabled = gl.is_enabled(gl::SCISSOR_TEST);
        unsafe {
            gl.get_integer_v(gl::DRAW_FRAMEBUFFER_BINDING, &mut bound_fbos[0..]);
            gl.get_integer_v(gl::READ_FRAMEBUFFER_BINDING, &mut bound_fbos[1..]);
            gl.get_float_v(gl::COLOR_CLEAR_VALUE, &mut clear_color[..]);
            gl.get_float_v(gl::DEPTH_CLEAR_VALUE, &mut clear_depth[..]);
            gl.get_integer_v(gl::STENCIL_CLEAR_VALUE, &mut clear_stencil[..]);
        }

        // Make the back buffer the current surface
        let reattach = match self.unattached_surface.take() {
            Some(surface) => Some(device.replace_context_surface(context, surface)?),
            None => None,
        };

        // Clear it
        let fbo = device.context_surface_framebuffer_object(context)?;
        gl.bind_framebuffer(gl::FRAMEBUFFER, fbo);
        gl.clear_color(0., 0., 0., 0.);
        gl.clear_depth(1.);
        gl.clear_stencil(0);
        gl.disable(gl::SCISSOR_TEST);
        gl.clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

        // Reattach the old surface
        if let Some(surface) = reattach {
            let surface = device.replace_context_surface(context, surface)?;
            self.unattached_surface = Some(surface);
        }

        // Restore the GL state
        gl.bind_framebuffer(gl::DRAW_FRAMEBUFFER, bound_fbos[0] as GLuint);
        gl.bind_framebuffer(gl::READ_FRAMEBUFFER, bound_fbos[1] as GLuint);
        gl.clear_color(
            clear_color[0],
            clear_color[1],
            clear_color[2],
            clear_color[3],
        );
        gl.clear_depth(clear_depth[0] as f64);
        gl.clear_stencil(clear_stencil[0]);
        if scissor_enabled {
            gl.enable(gl::SCISSOR_TEST);
        }

        Ok(())
    }

    // Destroy the swap chain.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for this swap chain.
    fn destroy(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.validate_context(device, context)?;
        let surfaces = self
            .pending_surface
            .take()
            .into_iter()
            .chain(self.unattached_surface.take().into_iter())
            .chain(self.recycled_surfaces.drain(..));
        for surface in surfaces {
            device.destroy_surface(context, surface)?;
        }
        Ok(())
    }
}

/// A thread-safe swap chain.
#[derive(Clone)]
pub struct SwapChain(Arc<Mutex<SwapChainData>>);

impl SwapChain {
    // Guarantee unique access to the swap chain data
    fn lock(&self) -> MutexGuard<SwapChainData> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }

    /// Swap the back and front buffers.
    /// Called by the producer.
    /// Returns an error if `context` is not the producer context for this swap chain.
    pub fn swap_buffers(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().swap_buffers(device, context)
    }

    /// Swap the attached swap chain.
    /// Called by the producer.
    /// Returns an error if `context` is not the producer context for both swap chains.
    /// Returns an error if this swap chain is attached, or the other swap chain is detached.
    pub fn take_attachment_from(
        &self,
        device: &mut Device,
        context: &mut Context,
        other: &SwapChain,
    ) -> Result<(), Error> {
        self.lock()
            .take_attachment_from(device, context, &mut *other.lock())
    }

    /// Resize the swap chain.
    /// This creates a new back buffer of the appropriate size,
    /// and destroys the old one.
    /// Called by the producer.
    /// Returns an error if `context` is not the producer context for this swap chain.
    pub fn resize(
        &self,
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        self.lock().resize(device, context, size)
    }

    // Clear the current back buffer.
    // Called by the producer.
    // Returns an error if `context` is not the producer context for this swap chain.
    pub fn clear_surface(
        &self,
        device: &mut Device,
        context: &mut Context,
        gl: &Gl,
    ) -> Result<(), Error> {
        self.lock().clear_surface(device, context, gl)
    }

    /// Take the current front buffer.
    /// Called by a consumer.
    pub fn take_surface(&self) -> Option<Surface> {
        self.lock().take_surface()
    }

    /// Recycle the current front buffer.
    /// Called by a consumer.
    pub fn recycle_surface(&self, surface: Surface) {
        self.lock().recycle_surface(surface)
    }

    /// Is this the attached swap chain?
    pub fn is_attached(&self) -> bool {
        self.lock().unattached_surface.is_none()
    }

    /// Destroy the swap chain.
    /// Called by the producer.
    /// Returns an error if `context` is not the producer context for this swap chain.
    pub fn destroy(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().destroy(device, context)
    }

    /// Create a new attached swap chain
    pub fn create_attached(
        device: &mut Device,
        context: &mut Context,
        surface_access: SurfaceAccess,
    ) -> Result<SwapChain, Error> {
        let size = device.context_surface_size(context)?;
        Ok(SwapChain(Arc::new(Mutex::new(SwapChainData {
            size,
            context_id: device.context_id(context),
            surface_access,
            unattached_surface: None,
            pending_surface: None,
            recycled_surfaces: Vec::new(),
        }))))
    }

    /// Create a new detached swap chain
    pub fn create_detached(
        device: &mut Device,
        context: &mut Context,
        surface_access: SurfaceAccess,
        size: Size2D<i32>,
    ) -> Result<SwapChain, Error> {
        let surface_type = SurfaceType::Generic { size };
        let surface = device.create_surface(context, surface_access, &surface_type)?;
        Ok(SwapChain(Arc::new(Mutex::new(SwapChainData {
            size,
            context_id: device.context_id(context),
            surface_access,
            unattached_surface: Some(surface),
            pending_surface: None,
            recycled_surfaces: Vec::new(),
        }))))
    }
}

/// A thread-safe collection of swap chains.
#[derive(Clone, Default)]
pub struct SwapChains<SwapChainID: Eq + Hash> {
    // The swap chain ids, indexed by context id
    ids: Arc<Mutex<FnvHashMap<ContextID, FnvHashSet<SwapChainID>>>>,
    // The swap chains, indexed by swap chain id
    table: Arc<RwLock<FnvHashMap<SwapChainID, SwapChain>>>,
}

impl<SwapChainID: Clone + Eq + Hash + Debug> SwapChains<SwapChainID> {
    /// Create a new collection.
    pub fn new() -> SwapChains<SwapChainID> {
        SwapChains {
            ids: Arc::new(Mutex::new(FnvHashMap::default())),
            table: Arc::new(RwLock::new(FnvHashMap::default())),
        }
    }

    // Lock the ids
    fn ids(&self) -> MutexGuard<FnvHashMap<ContextID, FnvHashSet<SwapChainID>>> {
        self.ids.lock().unwrap_or_else(|err| err.into_inner())
    }

    // Lock the lookup table
    fn table(&self) -> RwLockReadGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.read().unwrap_or_else(|err| err.into_inner())
    }

    // Lock the lookup table for writing
    fn table_mut(&self) -> RwLockWriteGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.write().unwrap_or_else(|err| err.into_inner())
    }

    /// Get a swap chain
    pub fn get(&self, id: SwapChainID) -> Option<SwapChain> {
        debug!("Getting swap chain {:?}", id);
        self.table().get(&id).cloned()
    }

    /// Create a new attached swap chain and insert it in the table.
    /// Returns an error if the `id` is already in the table.
    pub fn create_attached_swap_chain(
        &self,
        id: SwapChainID,
        device: &mut Device,
        context: &mut Context,
        surface_access: SurfaceAccess,
    ) -> Result<(), Error> {
        match self.table_mut().entry(id.clone()) {
            Entry::Occupied(_) => Err(Error::Failed)?,
            Entry::Vacant(entry) => {
                entry.insert(SwapChain::create_attached(device, context, surface_access)?)
            }
        };
        self.ids()
            .entry(device.context_id(context))
            .or_insert_with(Default::default)
            .insert(id);
        Ok(())
    }

    /// Create a new dettached swap chain and insert it in the table.
    /// Returns an error if the `id` is already in the table.
    pub fn create_detached_swap_chain(
        &self,
        id: SwapChainID,
        size: Size2D<i32>,
        device: &mut Device,
        context: &mut Context,
        surface_access: SurfaceAccess,
    ) -> Result<(), Error> {
        match self.table_mut().entry(id.clone()) {
            Entry::Occupied(_) => Err(Error::Failed)?,
            Entry::Vacant(entry) => entry.insert(SwapChain::create_detached(
                device,
                context,
                surface_access,
                size,
            )?),
        };
        self.ids()
            .entry(device.context_id(context))
            .or_insert_with(Default::default)
            .insert(id);
        Ok(())
    }

    /// Destroy a swap chain.
    /// Called by the producer.
    /// Returns an error if `context` is not the producer context for the swap chain.
    pub fn destroy(
        &self,
        id: SwapChainID,
        device: &mut Device,
        context: &mut Context,
    ) -> Result<(), Error> {
        if let Some(swap_chain) = self.table_mut().remove(&id) {
            swap_chain.destroy(device, context)?;
        }
        if let Some(ids) = self.ids().get_mut(&device.context_id(context)) {
            ids.remove(&id);
        }
        Ok(())
    }

    /// Destroy all the swap chains for a particular producer context.
    /// Called by the producer.
    pub fn destroy_all(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        if let Some(mut ids) = self.ids().remove(&device.context_id(context)) {
            for id in ids.drain() {
                if let Some(swap_chain) = self.table_mut().remove(&id) {
                    swap_chain.destroy(device, context)?;
                }
            }
        }
        Ok(())
    }

    /// Iterate over all the swap chains for a particular producer context.
    /// Called by the producer.
    pub fn iter(
        &self,
        device: &mut Device,
        context: &mut Context,
    ) -> impl Iterator<Item = (SwapChainID, SwapChain)> {
        self.ids()
            .get(&device.context_id(context))
            .iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| Some((id.clone(), self.table().get(id)?.clone())))
            .collect::<Vec<_>>()
            .into_iter()
    }
}
