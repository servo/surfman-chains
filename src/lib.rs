/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use euclid::default::Size2D;

use fnv::FnvHashMap;
use fnv::FnvHashSet;

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
        self.validate_context(context)?;

        // Fetch a new back buffer, recycling presented buffers if possible.
        let new_back_buffer = self
            .presented_surfaces
            .iter()
            .position(|surface| surface.size() == self.size)
            .map(|index| Ok(self.presented_surfaces.swap_remove(index)))
            .unwrap_or_else(|| device.create_surface(context, &self.size))?;

        // Swap the buffers
        let new_front_buffer = match self.unattached_front_buffer.as_mut() {
            Some(surface) => mem::replace(surface, new_back_buffer),
            None => device.replace_context_surface(context, new_back_buffer)?,
        };

        // Updata the state
        self.pending_surface = Some(new_front_buffer);
        for surface in self.presented_surfaces.drain(..) {
            device.destroy_surface(context, surface)?;
        }

        Ok(())
    }

    fn attach(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.validate_context(context)?;
        if let Some(surface) = self.unattached_front_buffer.take() {
            let surface = device.replace_context_surface(context, surface)?;
            device.destroy_surface(context, surface)?;
        }
        Ok(())
    }

    fn detach(&mut self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.validate_context(context)?;
        if self.unattached_front_buffer.is_none() {
            let surface = device.create_surface(context, &self.size)?;
            self.unattached_front_buffer = Some(surface);
        }
        Ok(())
    }

    fn resize(&mut self, device: &mut Device, context: &mut Context, size: Size2D<i32>) -> Result<(), Error> {
        self.validate_context(context)?;
        if let Some(surface) = self.unattached_front_buffer.as_mut() {
            let new_surface = device.create_surface(context, &size)?;
	    let old_surface = mem::replace(surface, new_surface);
            device.destroy_surface(context, old_surface)?;
        }
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

    pub fn attach(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().attach(device, context)
    }

    pub fn detach(&self, device: &mut Device, context: &mut Context) -> Result<(), Error> {
        self.lock().detach(device, context)
    }

    pub fn resize(&self, device: &mut Device, context: &mut Context, size: Size2D<i32>) -> Result<(), Error> {
        self.lock().resize(device, context, size)
    }

    pub fn take_surface(&self) -> Option<Surface> {
        self.lock().take_surface()
    }

    pub fn recycle_surface(&self, surface: Surface) {
        self.lock().recycle_surface(surface)
    }

    fn destroy(&self, device: &mut Device, context: &mut Context) {
        self.lock().destroy(device, context);
    }
}

#[derive(Clone, Default)]
pub struct SwapChains<SwapChainID: Eq + Hash> {
    ids: Arc<Mutex<FnvHashMap<ContextID, FnvHashSet<SwapChainID>>>>,
    table: Arc<RwLock<FnvHashMap<SwapChainID, SwapChain>>>,
}

impl<SwapChainID: Clone + Eq + Hash> SwapChains<SwapChainID> {
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
        self.table().get(&id).cloned()
    }

    pub fn get_with<F, T>(&self, id: SwapChainID, f: F) -> Option<T>
    where
        F: Fn(&SwapChain) -> T,
    {
        self.table().get(&id).map(f)
    }

    pub fn get_or_default(
        &self,
        id: SwapChainID,
        device: &Device,
        context: &mut Context,
    ) -> SwapChain {
        self.ids()
            .entry(context.id())
            .or_insert_with(Default::default)
            .insert(id.clone());
        self.table_mut()
            .entry(id)
            .or_insert_with(move || {
                let size = device.context_surface_size(context).unwrap();
                SwapChain(Arc::new(Mutex::new(SwapChainData {
                    size,
                    context_id: context.id(),
                    unattached_front_buffer: None,
                    pending_surface: None,
                    presented_surfaces: Vec::new(),
                })))
            })
            .clone()
    }

    pub fn get_or_create(
        &self,
        id: SwapChainID,
        device: &mut Device,
        context: &mut Context,
        size: Size2D<i32>,
    ) -> SwapChain {
        self.ids()
            .entry(context.id())
            .or_insert_with(Default::default)
            .insert(id.clone());
        self.table_mut()
            .entry(id)
            .or_insert_with(move || {
                let surface = device.create_surface(context, &size).unwrap();
                SwapChain(Arc::new(Mutex::new(SwapChainData {
                    size,
                    context_id: context.id(),
                    unattached_front_buffer: Some(surface),
                    pending_surface: None,
                    presented_surfaces: Vec::new(),
                })))
            })
            .clone()
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
}
