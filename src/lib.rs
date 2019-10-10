/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

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

use surfman::Context;
use surfman::ContextID;
use surfman::Device;
use surfman::Error;
use surfman::Surface;
use surfman::SurfaceType;

struct SwapChainData {
    size: Size2D<i32>,
    context_id: ContextID,
    unattached_front_buffer: Option<Surface>,
    pending_surface: Option<Surface>,
    presented_surfaces: Vec<Surface>,
}

impl SwapChainData {
    fn validate_context(&self, context: &mut Context) -> Result<(), Error> {
        if self.context_id == context.id() {
            Ok(())
        } else {
            Err(Error::IncompatibleContext)
        }
    }

    fn swap_buffers(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        debug!("Swap buffers on context {:?}", self.context_id);
        self.validate_context(context)?;

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
            .presented_surfaces
            .iter()
            .position(|surface| surface.size() == self.size)
            .map(|index| {
                debug!("Recyling surface for context {:?}", self.context_id);
                Ok(self.presented_surfaces.swap_remove(index))
            })
            .unwrap_or_else(|| {
                debug!(
                    "Creating a new surface ({:?}) for context {:?}",
                    self.size, self.context_id
                );
                let surface_type = SurfaceType::Generic { size: self.size };
                device.create_surface(context, &surface_type)
            })?;

        // Swap the buffers
        debug!(
            "Surface {:?} is the new back buffer for context {:?}",
            new_back_buffer.id(),
            self.context_id
        );
        let new_front_buffer = match self.unattached_front_buffer.as_mut() {
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
        for surface in self.presented_surfaces.drain(..) {
            debug!("Destroying a surface for context {:?}", self.context_id);
            device.destroy_surface(context, surface)?;
        }

        Ok(())
    }

    fn take_attachment_from(
        &mut self,
        device: &mut Device,
        context: &mut Context,
        other: &mut SwapChainData,
    ) -> Result<(), Error> {
        self.validate_context(context)?;
        other.validate_context(context)?;
        if let (Some(surface), true) = (
            self.unattached_front_buffer.take(),
            other.unattached_front_buffer.is_none(),
        ) {
            debug!("Attaching surface {:?}", surface.id());
            let surface = device.replace_context_surface(context, surface)?;
            debug!("Detaching surface {:?}", surface.id());
            other.unattached_front_buffer = Some(surface);
            Ok(())
        } else {
            Err(Error::Failed)
        }
    }

    fn resize(
        &mut self,
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        debug!("Resizing context {:?} to {:?}", context.id(), size);
        self.validate_context(context)?;
        let surface_type = SurfaceType::Generic { size };
        let new_back_buffer = device.create_surface(context, &surface_type)?;
        debug!(
            "Surface {:?} is the new back buffer for context {:?}",
            new_back_buffer.id(),
            self.context_id
        );
        let old_back_buffer = match self.unattached_front_buffer.as_mut() {
            Some(surface) => mem::replace(surface, new_back_buffer),
            None => device.replace_context_surface(context, new_back_buffer)?,
        };
        device.destroy_surface(context, old_back_buffer)?;
        self.size = size;
        Ok(())
    }

    fn take_surface(&mut self) -> Option<Surface> {
        self.pending_surface
            .take()
            .or_else(|| self.presented_surfaces.pop())
    }

    fn recycle_surface(&mut self, surface: Surface) {
        self.presented_surfaces.push(surface)
    }

    fn destroy(&mut self, device: &mut Device, context: &mut Context) {
        let surfaces = self
            .pending_surface
            .take()
            .into_iter()
            .chain(self.unattached_front_buffer.take().into_iter())
            .chain(self.presented_surfaces.drain(..));
        for surface in surfaces {
            device.destroy_surface(context, surface).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct SwapChain(Arc<Mutex<SwapChainData>>);

impl SwapChain {
    fn lock(&self) -> MutexGuard<SwapChainData> {
        self.0.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn swap_buffers(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().swap_buffers(device, context)
    }

    pub fn take_attachment_from(
        &self,
        device: &mut Device,
        context: &mut Context,
        other: &SwapChain,
    ) -> Result<(), Error> {
        self.lock()
            .take_attachment_from(device, context, &mut *other.lock())
    }

    pub fn resize(
        &self,
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<(), Error> {
        self.lock().resize(device, context, size)
    }

    pub fn take_surface(&self) -> Option<Surface> {
        self.lock().take_surface()
    }

    pub fn recycle_surface(&self, surface: Surface) {
        self.lock().recycle_surface(surface)
    }

    pub fn is_attached(&self) -> bool {
        self.lock().unattached_front_buffer.is_none()
    }

    fn destroy(&self, device: &mut Device, context: &mut Context) {
        self.lock().destroy(device, context);
    }

    fn create_attached(device: &mut Device, context: &mut Context) -> Result<SwapChain, Error> {
        let size = device.context_surface_size(context)?;
        Ok(SwapChain(Arc::new(Mutex::new(SwapChainData {
            size,
            context_id: context.id(),
            unattached_front_buffer: None,
            pending_surface: None,
            presented_surfaces: Vec::new(),
        }))))
    }

    fn create_detached(
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> Result<SwapChain, Error> {
        let surface_type = SurfaceType::Generic { size };
        let surface = device.create_surface(context, &surface_type)?;
        Ok(SwapChain(Arc::new(Mutex::new(SwapChainData {
            size,
            context_id: context.id(),
            unattached_front_buffer: Some(surface),
            pending_surface: None,
            presented_surfaces: Vec::new(),
        }))))
    }
}

#[derive(Clone, Default)]
pub struct SwapChains<SwapChainID: Eq + Hash> {
    ids: Arc<Mutex<FnvHashMap<ContextID, FnvHashSet<SwapChainID>>>>,
    table: Arc<RwLock<FnvHashMap<SwapChainID, SwapChain>>>,
}

impl<SwapChainID: Clone + Eq + Hash + Debug> SwapChains<SwapChainID> {
    pub fn new() -> SwapChains<SwapChainID> {
        SwapChains {
            ids: Arc::new(Mutex::new(FnvHashMap::default())),
            table: Arc::new(RwLock::new(FnvHashMap::default())),
        }
    }

    fn ids(&self) -> MutexGuard<FnvHashMap<ContextID, FnvHashSet<SwapChainID>>> {
        self.ids.lock().unwrap_or_else(|err| err.into_inner())
    }

    fn table(&self) -> RwLockReadGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.read().unwrap_or_else(|err| err.into_inner())
    }

    fn table_mut(&self) -> RwLockWriteGuard<FnvHashMap<SwapChainID, SwapChain>> {
        self.table.write().unwrap_or_else(|err| err.into_inner())
    }

    pub fn get(&self, id: SwapChainID) -> Option<SwapChain> {
        debug!("Getting swap chain {:?}", id);
        self.table().get(&id).cloned()
    }

    pub fn create_attached_swap_chain(
        &self,
        id: SwapChainID,
        device: &mut Device,
        context: &mut Context,
    ) -> Result<(), Error> {
        match self.table_mut().entry(id.clone()) {
            Entry::Occupied(_) => Err(Error::Failed)?,
            Entry::Vacant(entry) => entry.insert(SwapChain::create_attached(device, context)?),
        };
        self.ids()
            .entry(context.id())
            .or_insert_with(Default::default)
            .insert(id);
        Ok(())
    }

    pub fn create_detached_swap_chain(
        &self,
        id: SwapChainID,
        size: Size2D<i32>,
        device: &mut Device,
        context: &mut Context,
    ) -> Result<(), Error> {
        match self.table_mut().entry(id.clone()) {
            Entry::Occupied(_) => Err(Error::Failed)?,
            Entry::Vacant(entry) => {
                entry.insert(SwapChain::create_detached(device, context, size)?)
            }
        };
        self.ids()
            .entry(context.id())
            .or_insert_with(Default::default)
            .insert(id);
        Ok(())
    }

    pub fn destroy(&self, id: SwapChainID, device: &mut Device, context: &mut Context) {
        if let Some(swap_chain) = self.table_mut().remove(&id) {
            swap_chain.destroy(device, context);
        }
    }

    pub fn destroy_all(&self, device: &mut Device, context: &mut Context) {
        if let Some(mut ids) = self.ids().remove(&context.id()) {
            for id in ids.drain() {
                self.destroy(id, device, context);
            }
        }
    }

    pub fn iter(
        &self,
        _: &mut Device,
        context: &mut Context,
    ) -> impl Iterator<Item = (SwapChainID, SwapChain)> {
        self.ids()
            .get(&context.id())
            .iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| Some((id.clone(), self.table().get(id)?.clone())))
            .collect::<Vec<_>>()
            .into_iter()
    }
}
